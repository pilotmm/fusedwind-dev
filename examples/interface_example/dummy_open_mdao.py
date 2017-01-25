
class Component(object):

	def __init__(self):

		pass

	def add_param(self, **kwargs):

		print('OPENMDAO: Adding parameter', kwargs['name'], 'with the following attributes:')
		print(kwargs)

	def add_output(self, **kwargs):

		print('OPENMDAO: Adding output', kwargs['name'], 'with the following attributes:')
		print(kwargs)

