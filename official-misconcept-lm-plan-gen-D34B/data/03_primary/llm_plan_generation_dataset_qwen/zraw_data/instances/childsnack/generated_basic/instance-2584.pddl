; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 529565

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child313 - child
    bread377 - bread-portion
    content460 - content-portion
    tray471 tray391 tray273 - tray
    table92 table348 table123 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray471 kitchen)
     (at tray391 kitchen)
     (at tray273 kitchen)
     (at_kitchen_bread bread377)
     (at_kitchen_content content460)
     (not_allergic_gluten child313)
     (waiting child313 table348)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child313)
    )
  )
)
