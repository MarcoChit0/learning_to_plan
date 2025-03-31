; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 156194

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child177 child410 - child
    bread465 bread22 - bread-portion
    content322 content255 - content-portion
    tray424 tray335 - tray
    table129 table45 table305 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray424 kitchen)
     (at tray335 kitchen)
     (at_kitchen_bread bread465)
     (at_kitchen_bread bread22)
     (at_kitchen_content content322)
     (at_kitchen_content content255)
     (no_gluten_bread bread465)
     (no_gluten_content content255)
     (allergic_gluten child177)
     (not_allergic_gluten child410)
     (waiting child177 table305)
     (waiting child410 table45)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child177)
     (served child410)
    )
  )
)
