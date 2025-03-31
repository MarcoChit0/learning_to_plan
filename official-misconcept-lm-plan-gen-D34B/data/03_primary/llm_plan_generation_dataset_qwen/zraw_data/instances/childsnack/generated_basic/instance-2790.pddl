; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 533975

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child224 child294 - child
    bread394 bread247 - bread-portion
    content164 content469 - content-portion
    tray13 tray63 - tray
    table260 table140 table6 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray13 kitchen)
     (at tray63 kitchen)
     (at_kitchen_bread bread394)
     (at_kitchen_bread bread247)
     (at_kitchen_content content164)
     (at_kitchen_content content469)
     (not_allergic_gluten child224)
     (not_allergic_gluten child294)
     (waiting child224 table6)
     (waiting child294 table6)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child224)
     (served child294)
    )
  )
)
