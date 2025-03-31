; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 573903

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child38 child440 - child
    bread255 bread160 - bread-portion
    content207 content402 - content-portion
    tray116 tray421 tray19 - tray
    table79 table280 table378 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray116 kitchen)
     (at tray421 kitchen)
     (at tray19 kitchen)
     (at_kitchen_bread bread255)
     (at_kitchen_bread bread160)
     (at_kitchen_content content207)
     (at_kitchen_content content402)
     (not_allergic_gluten child38)
     (not_allergic_gluten child440)
     (waiting child38 table79)
     (waiting child440 table378)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child38)
     (served child440)
    )
  )
)
