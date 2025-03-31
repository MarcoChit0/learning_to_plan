; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 404775

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child42 - child
    bread498 - bread-portion
    content383 - content-portion
    tray490 tray451 tray369 - tray
    table15 table387 table159 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray490 kitchen)
     (at tray451 kitchen)
     (at tray369 kitchen)
     (at_kitchen_bread bread498)
     (at_kitchen_content content383)
     (not_allergic_gluten child42)
     (waiting child42 table387)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child42)
    )
  )
)
