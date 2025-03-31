; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 570236

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child468 child435 - child
    bread129 bread231 - bread-portion
    content331 content211 - content-portion
    tray482 tray89 tray352 tray7 - tray
    table319 table220 table0 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray482 kitchen)
     (at tray89 kitchen)
     (at tray352 kitchen)
     (at tray7 kitchen)
     (at_kitchen_bread bread129)
     (at_kitchen_bread bread231)
     (at_kitchen_content content331)
     (at_kitchen_content content211)
     (not_allergic_gluten child468)
     (not_allergic_gluten child435)
     (waiting child468 table0)
     (waiting child435 table0)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child468)
     (served child435)
    )
  )
)
