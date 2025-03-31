; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 399941

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child114 - child
    bread265 - bread-portion
    content74 - content-portion
    tray329 tray429 tray60 - tray
    table320 table202 table404 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray329 kitchen)
     (at tray429 kitchen)
     (at tray60 kitchen)
     (at_kitchen_bread bread265)
     (at_kitchen_content content74)
     (not_allergic_gluten child114)
     (waiting child114 table404)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child114)
    )
  )
)
