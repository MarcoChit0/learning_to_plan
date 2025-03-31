; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 192472

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child114 child448 - child
    bread476 bread294 - bread-portion
    content156 content57 - content-portion
    tray259 tray121 tray131 - tray
    table39 table405 table57 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray259 kitchen)
     (at tray121 kitchen)
     (at tray131 kitchen)
     (at_kitchen_bread bread476)
     (at_kitchen_bread bread294)
     (at_kitchen_content content156)
     (at_kitchen_content content57)
     (not_allergic_gluten child114)
     (not_allergic_gluten child448)
     (waiting child114 table57)
     (waiting child448 table405)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child114)
     (served child448)
    )
  )
)
