; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 372974

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child296 - child
    bread115 - bread-portion
    content435 - content-portion
    tray91 tray433 - tray
    table16 table285 table359 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray91 kitchen)
     (at tray433 kitchen)
     (at_kitchen_bread bread115)
     (at_kitchen_content content435)
     (not_allergic_gluten child296)
     (waiting child296 table359)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child296)
    )
  )
)
