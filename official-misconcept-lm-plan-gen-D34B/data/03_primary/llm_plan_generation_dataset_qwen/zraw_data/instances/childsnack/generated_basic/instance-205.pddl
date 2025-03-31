; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 224592

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child367 child167 - child
    bread251 bread80 - bread-portion
    content121 content322 - content-portion
    tray329 tray43 tray199 - tray
    table162 table258 table241 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray329 kitchen)
     (at tray43 kitchen)
     (at tray199 kitchen)
     (at_kitchen_bread bread251)
     (at_kitchen_bread bread80)
     (at_kitchen_content content121)
     (at_kitchen_content content322)
     (not_allergic_gluten child367)
     (not_allergic_gluten child167)
     (waiting child367 table241)
     (waiting child167 table258)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child367)
     (served child167)
    )
  )
)
