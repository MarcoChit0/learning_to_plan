; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 76035

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child261 - child
    bread166 - bread-portion
    content110 - content-portion
    tray160 tray50 tray25 tray236 - tray
    table462 table58 table444 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray160 kitchen)
     (at tray50 kitchen)
     (at tray25 kitchen)
     (at tray236 kitchen)
     (at_kitchen_bread bread166)
     (at_kitchen_content content110)
     (not_allergic_gluten child261)
     (waiting child261 table58)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child261)
    )
  )
)
