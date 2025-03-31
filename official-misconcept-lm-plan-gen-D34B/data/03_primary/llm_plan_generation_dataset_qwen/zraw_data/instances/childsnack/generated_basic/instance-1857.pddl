; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 287743

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child64 child497 - child
    bread349 bread325 - bread-portion
    content116 content11 - content-portion
    tray448 tray439 - tray
    table32 table252 table277 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray448 kitchen)
     (at tray439 kitchen)
     (at_kitchen_bread bread349)
     (at_kitchen_bread bread325)
     (at_kitchen_content content116)
     (at_kitchen_content content11)
     (no_gluten_bread bread325)
     (no_gluten_content content116)
     (allergic_gluten child497)
     (not_allergic_gluten child64)
     (waiting child64 table32)
     (waiting child497 table252)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child64)
     (served child497)
    )
  )
)
