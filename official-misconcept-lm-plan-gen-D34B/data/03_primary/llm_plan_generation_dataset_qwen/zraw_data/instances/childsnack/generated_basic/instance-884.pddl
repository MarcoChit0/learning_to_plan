; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 60350

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child323 child18 child352 - child
    bread144 bread346 bread418 - bread-portion
    content413 content403 content410 - content-portion
    tray150 - tray
    table106 table40 table334 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray150 kitchen)
     (at_kitchen_bread bread144)
     (at_kitchen_bread bread346)
     (at_kitchen_bread bread418)
     (at_kitchen_content content413)
     (at_kitchen_content content403)
     (at_kitchen_content content410)
     (not_allergic_gluten child352)
     (not_allergic_gluten child18)
     (not_allergic_gluten child323)
     (waiting child323 table334)
     (waiting child18 table106)
     (waiting child352 table334)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child323)
     (served child18)
     (served child352)
    )
  )
)
