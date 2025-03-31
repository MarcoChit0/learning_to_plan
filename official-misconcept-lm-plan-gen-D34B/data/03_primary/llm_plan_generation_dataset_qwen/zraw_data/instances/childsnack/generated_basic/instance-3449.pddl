; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 450788

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child305 - child
    bread137 - bread-portion
    content414 - content-portion
    tray469 tray92 tray202 tray16 - tray
    table228 table487 table257 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray469 kitchen)
     (at tray92 kitchen)
     (at tray202 kitchen)
     (at tray16 kitchen)
     (at_kitchen_bread bread137)
     (at_kitchen_content content414)
     (not_allergic_gluten child305)
     (waiting child305 table257)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child305)
    )
  )
)
