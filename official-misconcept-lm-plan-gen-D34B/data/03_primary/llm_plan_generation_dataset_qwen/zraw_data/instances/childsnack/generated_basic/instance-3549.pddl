; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 96655

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child211 child124 - child
    bread45 bread264 - bread-portion
    content268 content210 - content-portion
    tray476 tray202 - tray
    table381 table48 table319 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray476 kitchen)
     (at tray202 kitchen)
     (at_kitchen_bread bread45)
     (at_kitchen_bread bread264)
     (at_kitchen_content content268)
     (at_kitchen_content content210)
     (not_allergic_gluten child124)
     (not_allergic_gluten child211)
     (waiting child211 table48)
     (waiting child124 table319)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child211)
     (served child124)
    )
  )
)
