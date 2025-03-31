; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 144226

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child409 - child
    bread155 - bread-portion
    content155 - content-portion
    tray209 tray318 tray323 tray262 - tray
    table209 table424 table142 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray209 kitchen)
     (at tray318 kitchen)
     (at tray323 kitchen)
     (at tray262 kitchen)
     (at_kitchen_bread bread155)
     (at_kitchen_content content155)
     (not_allergic_gluten child409)
     (waiting child409 table424)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child409)
    )
  )
)
