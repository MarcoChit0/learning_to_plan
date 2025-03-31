; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 743930

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child286 child339 child270 - child
    bread392 bread332 bread308 - bread-portion
    content386 content129 content420 - content-portion
    tray83 - tray
    table113 table499 table132 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray83 kitchen)
     (at_kitchen_bread bread392)
     (at_kitchen_bread bread332)
     (at_kitchen_bread bread308)
     (at_kitchen_content content386)
     (at_kitchen_content content129)
     (at_kitchen_content content420)
     (not_allergic_gluten child270)
     (not_allergic_gluten child286)
     (not_allergic_gluten child339)
     (waiting child286 table113)
     (waiting child339 table113)
     (waiting child270 table113)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child286)
     (served child339)
     (served child270)
    )
  )
)
