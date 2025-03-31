; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 299668

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child323 - child
    bread184 - bread-portion
    content417 - content-portion
    tray230 tray69 tray442 tray357 - tray
    table271 table109 table301 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray230 kitchen)
     (at tray69 kitchen)
     (at tray442 kitchen)
     (at tray357 kitchen)
     (at_kitchen_bread bread184)
     (at_kitchen_content content417)
     (not_allergic_gluten child323)
     (waiting child323 table301)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child323)
    )
  )
)
