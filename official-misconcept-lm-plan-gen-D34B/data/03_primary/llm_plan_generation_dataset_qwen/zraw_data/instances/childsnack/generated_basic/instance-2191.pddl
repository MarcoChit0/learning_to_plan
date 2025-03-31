; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 133598

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child437 - child
    bread434 - bread-portion
    content461 - content-portion
    tray136 - tray
    table44 table302 table38 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray136 kitchen)
     (at_kitchen_bread bread434)
     (at_kitchen_content content461)
     (not_allergic_gluten child437)
     (waiting child437 table302)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child437)
    )
  )
)
