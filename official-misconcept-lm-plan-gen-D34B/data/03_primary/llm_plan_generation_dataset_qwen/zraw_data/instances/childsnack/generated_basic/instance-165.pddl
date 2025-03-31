; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 615104

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child158 child328 - child
    bread285 bread178 - bread-portion
    content457 content421 - content-portion
    tray285 tray219 tray273 tray342 - tray
    table330 table414 table121 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray285 kitchen)
     (at tray219 kitchen)
     (at tray273 kitchen)
     (at tray342 kitchen)
     (at_kitchen_bread bread285)
     (at_kitchen_bread bread178)
     (at_kitchen_content content457)
     (at_kitchen_content content421)
     (not_allergic_gluten child328)
     (not_allergic_gluten child158)
     (waiting child158 table330)
     (waiting child328 table121)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child158)
     (served child328)
    )
  )
)
