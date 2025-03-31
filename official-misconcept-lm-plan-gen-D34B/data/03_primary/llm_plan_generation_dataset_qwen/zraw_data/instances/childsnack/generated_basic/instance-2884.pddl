; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 196674

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child196 - child
    bread227 - bread-portion
    content245 - content-portion
    tray188 tray136 - tray
    table320 table132 table459 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray188 kitchen)
     (at tray136 kitchen)
     (at_kitchen_bread bread227)
     (at_kitchen_content content245)
     (not_allergic_gluten child196)
     (waiting child196 table459)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child196)
    )
  )
)
