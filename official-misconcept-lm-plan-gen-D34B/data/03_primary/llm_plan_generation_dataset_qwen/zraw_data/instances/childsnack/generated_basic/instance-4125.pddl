; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 243633

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child331 child391 - child
    bread202 bread437 - bread-portion
    content155 content394 - content-portion
    tray340 tray293 tray13 - tray
    table110 table477 table56 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray340 kitchen)
     (at tray293 kitchen)
     (at tray13 kitchen)
     (at_kitchen_bread bread202)
     (at_kitchen_bread bread437)
     (at_kitchen_content content155)
     (at_kitchen_content content394)
     (not_allergic_gluten child331)
     (not_allergic_gluten child391)
     (waiting child331 table110)
     (waiting child391 table56)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child331)
     (served child391)
    )
  )
)
