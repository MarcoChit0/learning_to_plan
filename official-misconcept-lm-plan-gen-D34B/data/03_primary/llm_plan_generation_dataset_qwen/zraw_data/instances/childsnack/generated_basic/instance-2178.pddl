; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 201370

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child62 - child
    bread347 - bread-portion
    content385 - content-portion
    tray374 - tray
    table189 table80 table100 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray374 kitchen)
     (at_kitchen_bread bread347)
     (at_kitchen_content content385)
     (not_allergic_gluten child62)
     (waiting child62 table100)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child62)
    )
  )
)
