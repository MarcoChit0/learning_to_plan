; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 57317

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child157 child465 - child
    bread326 bread444 - bread-portion
    content169 content356 - content-portion
    tray36 tray429 tray67 - tray
    table37 table121 table61 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray36 kitchen)
     (at tray429 kitchen)
     (at tray67 kitchen)
     (at_kitchen_bread bread326)
     (at_kitchen_bread bread444)
     (at_kitchen_content content169)
     (at_kitchen_content content356)
     (not_allergic_gluten child465)
     (not_allergic_gluten child157)
     (waiting child157 table121)
     (waiting child465 table37)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child157)
     (served child465)
    )
  )
)
