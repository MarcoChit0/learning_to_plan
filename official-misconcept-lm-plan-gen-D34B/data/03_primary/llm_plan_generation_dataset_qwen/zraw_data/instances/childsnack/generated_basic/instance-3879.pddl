; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 745353

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child475 - child
    bread246 - bread-portion
    content463 - content-portion
    tray302 tray484 - tray
    table355 table477 table183 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray302 kitchen)
     (at tray484 kitchen)
     (at_kitchen_bread bread246)
     (at_kitchen_content content463)
     (not_allergic_gluten child475)
     (waiting child475 table477)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child475)
    )
  )
)
