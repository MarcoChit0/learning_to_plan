; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 893590

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child294 child204 - child
    bread83 bread352 - bread-portion
    content9 content364 - content-portion
    tray153 tray486 tray350 tray287 - tray
    table273 table291 table111 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray153 kitchen)
     (at tray486 kitchen)
     (at tray350 kitchen)
     (at tray287 kitchen)
     (at_kitchen_bread bread83)
     (at_kitchen_bread bread352)
     (at_kitchen_content content9)
     (at_kitchen_content content364)
     (not_allergic_gluten child294)
     (not_allergic_gluten child204)
     (waiting child294 table273)
     (waiting child204 table273)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child294)
     (served child204)
    )
  )
)
