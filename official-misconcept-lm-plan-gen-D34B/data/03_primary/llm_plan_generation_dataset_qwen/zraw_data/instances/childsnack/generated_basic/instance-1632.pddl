; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 304095

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child37 - child
    bread257 - bread-portion
    content109 - content-portion
    tray371 tray427 tray66 tray467 - tray
    table255 table296 table304 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray371 kitchen)
     (at tray427 kitchen)
     (at tray66 kitchen)
     (at tray467 kitchen)
     (at_kitchen_bread bread257)
     (at_kitchen_content content109)
     (not_allergic_gluten child37)
     (waiting child37 table255)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child37)
    )
  )
)
