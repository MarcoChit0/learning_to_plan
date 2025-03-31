; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 839556

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child52 - child
    bread323 - bread-portion
    content70 - content-portion
    tray398 tray87 tray282 tray190 - tray
    table220 table400 table162 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray398 kitchen)
     (at tray87 kitchen)
     (at tray282 kitchen)
     (at tray190 kitchen)
     (at_kitchen_bread bread323)
     (at_kitchen_content content70)
     (not_allergic_gluten child52)
     (waiting child52 table400)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child52)
    )
  )
)
