; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 854066

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child489 child287 - child
    bread34 bread333 - bread-portion
    content250 content150 - content-portion
    tray324 tray348 - tray
    table80 table128 table21 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray324 kitchen)
     (at tray348 kitchen)
     (at_kitchen_bread bread34)
     (at_kitchen_bread bread333)
     (at_kitchen_content content250)
     (at_kitchen_content content150)
     (no_gluten_bread bread333)
     (no_gluten_content content250)
     (allergic_gluten child287)
     (not_allergic_gluten child489)
     (waiting child489 table21)
     (waiting child287 table21)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child489)
     (served child287)
    )
  )
)
