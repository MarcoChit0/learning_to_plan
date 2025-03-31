; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 755020

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child109 child226 child347 - child
    bread72 bread30 bread219 - bread-portion
    content38 content234 content219 - content-portion
    tray302 tray335 tray175 - tray
    table359 table155 table366 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray302 kitchen)
     (at tray335 kitchen)
     (at tray175 kitchen)
     (at_kitchen_bread bread72)
     (at_kitchen_bread bread30)
     (at_kitchen_bread bread219)
     (at_kitchen_content content38)
     (at_kitchen_content content234)
     (at_kitchen_content content219)
     (not_allergic_gluten child109)
     (not_allergic_gluten child347)
     (not_allergic_gluten child226)
     (waiting child109 table359)
     (waiting child226 table366)
     (waiting child347 table359)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child109)
     (served child226)
     (served child347)
    )
  )
)
