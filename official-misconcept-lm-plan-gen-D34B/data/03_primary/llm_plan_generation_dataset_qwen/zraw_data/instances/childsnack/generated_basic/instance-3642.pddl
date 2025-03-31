; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 797864

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child56 child110 - child
    bread212 bread438 - bread-portion
    content57 content407 - content-portion
    tray383 tray365 tray155 tray306 - tray
    table295 table190 table46 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray383 kitchen)
     (at tray365 kitchen)
     (at tray155 kitchen)
     (at tray306 kitchen)
     (at_kitchen_bread bread212)
     (at_kitchen_bread bread438)
     (at_kitchen_content content57)
     (at_kitchen_content content407)
     (no_gluten_bread bread438)
     (no_gluten_content content57)
     (allergic_gluten child56)
     (not_allergic_gluten child110)
     (waiting child56 table295)
     (waiting child110 table190)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child56)
     (served child110)
    )
  )
)
