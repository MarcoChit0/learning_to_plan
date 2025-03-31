; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 576781

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child442 child355 - child
    bread388 bread465 - bread-portion
    content121 content308 - content-portion
    tray425 tray177 tray467 tray274 - tray
    table312 table18 table285 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray425 kitchen)
     (at tray177 kitchen)
     (at tray467 kitchen)
     (at tray274 kitchen)
     (at_kitchen_bread bread388)
     (at_kitchen_bread bread465)
     (at_kitchen_content content121)
     (at_kitchen_content content308)
     (not_allergic_gluten child355)
     (not_allergic_gluten child442)
     (waiting child442 table312)
     (waiting child355 table312)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child442)
     (served child355)
    )
  )
)
