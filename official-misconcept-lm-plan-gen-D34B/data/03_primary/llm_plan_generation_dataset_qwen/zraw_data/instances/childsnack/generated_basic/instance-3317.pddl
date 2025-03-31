; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 35444

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child440 child441 - child
    bread260 bread476 - bread-portion
    content147 content268 - content-portion
    tray310 tray373 tray493 tray347 - tray
    table338 table80 table8 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray310 kitchen)
     (at tray373 kitchen)
     (at tray493 kitchen)
     (at tray347 kitchen)
     (at_kitchen_bread bread260)
     (at_kitchen_bread bread476)
     (at_kitchen_content content147)
     (at_kitchen_content content268)
     (no_gluten_bread bread476)
     (no_gluten_content content147)
     (allergic_gluten child441)
     (not_allergic_gluten child440)
     (waiting child440 table80)
     (waiting child441 table338)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child440)
     (served child441)
    )
  )
)
