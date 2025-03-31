; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 469403

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child303 - child
    bread371 - bread-portion
    content170 - content-portion
    tray174 tray455 tray410 tray27 - tray
    table291 table334 table258 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray174 kitchen)
     (at tray455 kitchen)
     (at tray410 kitchen)
     (at tray27 kitchen)
     (at_kitchen_bread bread371)
     (at_kitchen_content content170)
     (not_allergic_gluten child303)
     (waiting child303 table334)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child303)
    )
  )
)
