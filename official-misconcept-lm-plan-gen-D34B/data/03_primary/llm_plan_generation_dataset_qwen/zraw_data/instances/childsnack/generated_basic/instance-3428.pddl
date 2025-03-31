; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 476766

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child287 - child
    bread382 - bread-portion
    content324 - content-portion
    tray496 tray488 tray400 tray163 - tray
    table62 table345 table257 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray496 kitchen)
     (at tray488 kitchen)
     (at tray400 kitchen)
     (at tray163 kitchen)
     (at_kitchen_bread bread382)
     (at_kitchen_content content324)
     (not_allergic_gluten child287)
     (waiting child287 table62)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child287)
    )
  )
)
