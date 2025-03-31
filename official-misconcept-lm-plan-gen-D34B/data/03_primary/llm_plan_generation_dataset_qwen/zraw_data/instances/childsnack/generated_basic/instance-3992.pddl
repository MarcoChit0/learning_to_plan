; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 732241

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child174 - child
    bread343 - bread-portion
    content468 - content-portion
    tray46 tray423 tray217 - tray
    table169 table279 table45 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray46 kitchen)
     (at tray423 kitchen)
     (at tray217 kitchen)
     (at_kitchen_bread bread343)
     (at_kitchen_content content468)
     (not_allergic_gluten child174)
     (waiting child174 table45)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child174)
    )
  )
)
