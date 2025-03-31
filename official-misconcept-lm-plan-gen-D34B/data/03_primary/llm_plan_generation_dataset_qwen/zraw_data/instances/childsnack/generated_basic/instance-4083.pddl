; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 908272

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child399 - child
    bread279 - bread-portion
    content26 - content-portion
    tray139 tray246 tray487 - tray
    table447 table105 table440 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray139 kitchen)
     (at tray246 kitchen)
     (at tray487 kitchen)
     (at_kitchen_bread bread279)
     (at_kitchen_content content26)
     (not_allergic_gluten child399)
     (waiting child399 table447)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child399)
    )
  )
)
