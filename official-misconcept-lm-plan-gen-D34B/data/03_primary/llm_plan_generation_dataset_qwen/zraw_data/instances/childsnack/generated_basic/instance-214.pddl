; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 790199

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child80 child155 - child
    bread190 bread159 - bread-portion
    content232 content353 - content-portion
    tray191 tray65 tray133 - tray
    table90 table166 table459 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray191 kitchen)
     (at tray65 kitchen)
     (at tray133 kitchen)
     (at_kitchen_bread bread190)
     (at_kitchen_bread bread159)
     (at_kitchen_content content232)
     (at_kitchen_content content353)
     (not_allergic_gluten child155)
     (not_allergic_gluten child80)
     (waiting child80 table166)
     (waiting child155 table90)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child80)
     (served child155)
    )
  )
)
