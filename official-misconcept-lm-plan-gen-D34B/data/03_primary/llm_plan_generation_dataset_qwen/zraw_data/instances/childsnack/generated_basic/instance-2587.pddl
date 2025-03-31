; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 940306

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child57 - child
    bread355 - bread-portion
    content498 - content-portion
    tray491 tray49 tray81 - tray
    table251 table191 table17 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray491 kitchen)
     (at tray49 kitchen)
     (at tray81 kitchen)
     (at_kitchen_bread bread355)
     (at_kitchen_content content498)
     (not_allergic_gluten child57)
     (waiting child57 table251)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child57)
    )
  )
)
