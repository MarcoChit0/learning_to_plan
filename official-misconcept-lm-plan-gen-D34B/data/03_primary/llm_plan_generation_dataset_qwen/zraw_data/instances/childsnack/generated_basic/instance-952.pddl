; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 418722

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child82 child133 child366 - child
    bread363 bread82 bread202 - bread-portion
    content425 content415 content229 - content-portion
    tray76 tray305 tray476 tray405 - tray
    table417 table474 table302 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray76 kitchen)
     (at tray305 kitchen)
     (at tray476 kitchen)
     (at tray405 kitchen)
     (at_kitchen_bread bread363)
     (at_kitchen_bread bread82)
     (at_kitchen_bread bread202)
     (at_kitchen_content content425)
     (at_kitchen_content content415)
     (at_kitchen_content content229)
     (not_allergic_gluten child82)
     (not_allergic_gluten child366)
     (not_allergic_gluten child133)
     (waiting child82 table302)
     (waiting child133 table302)
     (waiting child366 table302)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child82)
     (served child133)
     (served child366)
    )
  )
)
