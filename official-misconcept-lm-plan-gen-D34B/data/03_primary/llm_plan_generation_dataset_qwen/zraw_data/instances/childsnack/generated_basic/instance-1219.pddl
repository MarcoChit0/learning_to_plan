; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 813680

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child151 child406 child83 - child
    bread10 bread395 bread116 - bread-portion
    content266 content235 content328 - content-portion
    tray291 tray145 tray462 - tray
    table384 table338 table413 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray291 kitchen)
     (at tray145 kitchen)
     (at tray462 kitchen)
     (at_kitchen_bread bread10)
     (at_kitchen_bread bread395)
     (at_kitchen_bread bread116)
     (at_kitchen_content content266)
     (at_kitchen_content content235)
     (at_kitchen_content content328)
     (not_allergic_gluten child406)
     (not_allergic_gluten child83)
     (not_allergic_gluten child151)
     (waiting child151 table413)
     (waiting child406 table338)
     (waiting child83 table413)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child151)
     (served child406)
     (served child83)
    )
  )
)
