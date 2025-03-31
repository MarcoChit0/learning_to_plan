; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 657478

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child490 child37 - child
    bread114 bread242 - bread-portion
    content490 content19 - content-portion
    tray320 tray449 - tray
    table287 table360 table9 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray320 kitchen)
     (at tray449 kitchen)
     (at_kitchen_bread bread114)
     (at_kitchen_bread bread242)
     (at_kitchen_content content490)
     (at_kitchen_content content19)
     (not_allergic_gluten child37)
     (not_allergic_gluten child490)
     (waiting child490 table360)
     (waiting child37 table287)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child490)
     (served child37)
    )
  )
)
