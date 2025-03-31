; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 518246

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child183 - child
    bread423 - bread-portion
    content169 - content-portion
    tray281 tray76 tray244 tray422 - tray
    table474 table190 table374 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray281 kitchen)
     (at tray76 kitchen)
     (at tray244 kitchen)
     (at tray422 kitchen)
     (at_kitchen_bread bread423)
     (at_kitchen_content content169)
     (not_allergic_gluten child183)
     (waiting child183 table190)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child183)
    )
  )
)
