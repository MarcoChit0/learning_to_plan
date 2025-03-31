; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 486144

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child469 child194 - child
    bread457 bread192 - bread-portion
    content96 content127 - content-portion
    tray116 tray236 tray245 - tray
    table134 table166 table282 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray116 kitchen)
     (at tray236 kitchen)
     (at tray245 kitchen)
     (at_kitchen_bread bread457)
     (at_kitchen_bread bread192)
     (at_kitchen_content content96)
     (at_kitchen_content content127)
     (not_allergic_gluten child194)
     (not_allergic_gluten child469)
     (waiting child469 table134)
     (waiting child194 table282)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child469)
     (served child194)
    )
  )
)
