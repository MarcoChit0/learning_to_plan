; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 609823

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child82 - child
    bread163 - bread-portion
    content465 - content-portion
    tray252 tray217 tray55 - tray
    table263 table211 table389 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray252 kitchen)
     (at tray217 kitchen)
     (at tray55 kitchen)
     (at_kitchen_bread bread163)
     (at_kitchen_content content465)
     (not_allergic_gluten child82)
     (waiting child82 table211)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child82)
    )
  )
)
