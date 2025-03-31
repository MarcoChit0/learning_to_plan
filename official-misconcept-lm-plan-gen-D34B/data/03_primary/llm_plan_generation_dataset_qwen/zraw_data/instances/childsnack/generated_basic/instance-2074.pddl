; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 33803

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child408 - child
    bread14 - bread-portion
    content432 - content-portion
    tray397 tray406 tray70 tray92 - tray
    table106 table89 table51 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray397 kitchen)
     (at tray406 kitchen)
     (at tray70 kitchen)
     (at tray92 kitchen)
     (at_kitchen_bread bread14)
     (at_kitchen_content content432)
     (not_allergic_gluten child408)
     (waiting child408 table106)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child408)
    )
  )
)
