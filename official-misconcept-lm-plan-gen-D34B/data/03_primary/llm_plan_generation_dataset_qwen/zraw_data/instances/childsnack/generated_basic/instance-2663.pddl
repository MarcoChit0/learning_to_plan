; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 265396

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child2 - child
    bread317 - bread-portion
    content14 - content-portion
    tray210 tray70 tray437 tray232 - tray
    table301 table12 table471 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray210 kitchen)
     (at tray70 kitchen)
     (at tray437 kitchen)
     (at tray232 kitchen)
     (at_kitchen_bread bread317)
     (at_kitchen_content content14)
     (not_allergic_gluten child2)
     (waiting child2 table301)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child2)
    )
  )
)
