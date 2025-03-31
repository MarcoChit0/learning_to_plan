; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 948896

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child43 child444 - child
    bread33 bread195 - bread-portion
    content475 content9 - content-portion
    tray39 tray489 - tray
    table103 table53 table375 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray39 kitchen)
     (at tray489 kitchen)
     (at_kitchen_bread bread33)
     (at_kitchen_bread bread195)
     (at_kitchen_content content475)
     (at_kitchen_content content9)
     (not_allergic_gluten child444)
     (not_allergic_gluten child43)
     (waiting child43 table53)
     (waiting child444 table375)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child43)
     (served child444)
    )
  )
)
