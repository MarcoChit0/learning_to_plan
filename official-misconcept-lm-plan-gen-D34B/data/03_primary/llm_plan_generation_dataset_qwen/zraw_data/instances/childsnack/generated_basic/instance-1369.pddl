; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 960771

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child150 child110 - child
    bread306 bread479 - bread-portion
    content430 content256 - content-portion
    tray251 tray51 tray477 - tray
    table201 table224 table435 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray251 kitchen)
     (at tray51 kitchen)
     (at tray477 kitchen)
     (at_kitchen_bread bread306)
     (at_kitchen_bread bread479)
     (at_kitchen_content content430)
     (at_kitchen_content content256)
     (not_allergic_gluten child150)
     (not_allergic_gluten child110)
     (waiting child150 table201)
     (waiting child110 table224)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child150)
     (served child110)
    )
  )
)
