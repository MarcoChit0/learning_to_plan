; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 159247

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child400 child448 - child
    bread35 bread116 - bread-portion
    content469 content124 - content-portion
    tray113 tray449 - tray
    table446 table492 table269 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray113 kitchen)
     (at tray449 kitchen)
     (at_kitchen_bread bread35)
     (at_kitchen_bread bread116)
     (at_kitchen_content content469)
     (at_kitchen_content content124)
     (not_allergic_gluten child448)
     (not_allergic_gluten child400)
     (waiting child400 table492)
     (waiting child448 table269)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child400)
     (served child448)
    )
  )
)
