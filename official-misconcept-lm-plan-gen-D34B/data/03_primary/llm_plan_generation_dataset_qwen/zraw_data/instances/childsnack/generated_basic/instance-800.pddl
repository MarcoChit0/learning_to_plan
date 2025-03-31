; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 884257

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child395 - child
    bread293 - bread-portion
    content379 - content-portion
    tray152 tray456 - tray
    table395 table396 table255 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray152 kitchen)
     (at tray456 kitchen)
     (at_kitchen_bread bread293)
     (at_kitchen_content content379)
     (not_allergic_gluten child395)
     (waiting child395 table396)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child395)
    )
  )
)
