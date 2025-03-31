; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 454224

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child16 child466 - child
    bread44 bread421 - bread-portion
    content343 content264 - content-portion
    tray21 tray86 - tray
    table338 table83 table274 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray21 kitchen)
     (at tray86 kitchen)
     (at_kitchen_bread bread44)
     (at_kitchen_bread bread421)
     (at_kitchen_content content343)
     (at_kitchen_content content264)
     (not_allergic_gluten child466)
     (not_allergic_gluten child16)
     (waiting child16 table274)
     (waiting child466 table274)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child16)
     (served child466)
    )
  )
)
