; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 488253

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child31 child278 - child
    bread298 bread87 - bread-portion
    content432 content374 - content-portion
    tray331 tray211 tray398 tray489 - tray
    table319 table466 table210 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray331 kitchen)
     (at tray211 kitchen)
     (at tray398 kitchen)
     (at tray489 kitchen)
     (at_kitchen_bread bread298)
     (at_kitchen_bread bread87)
     (at_kitchen_content content432)
     (at_kitchen_content content374)
     (not_allergic_gluten child278)
     (not_allergic_gluten child31)
     (waiting child31 table319)
     (waiting child278 table466)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child31)
     (served child278)
    )
  )
)
