; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 567153

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child36 - child
    bread400 - bread-portion
    content424 - content-portion
    tray168 tray88 - tray
    table334 table105 table142 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray168 kitchen)
     (at tray88 kitchen)
     (at_kitchen_bread bread400)
     (at_kitchen_content content424)
     (not_allergic_gluten child36)
     (waiting child36 table105)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child36)
    )
  )
)
